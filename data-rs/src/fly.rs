use std::env::var;
use crate::common::{ArchivedNode, ArchivedOffsets, ArchivedTableType, Offsets, SemType};
use clap::Parser;
use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::types::PyTuple;
use pyo3::{Bound, PyErr, Python};
use pyo3::{IntoPyObject, pyclass, pymethods};
use rand::prelude::*;
use rkyv::rancor::Error;
use std::collections::VecDeque;
use std::fs;
use std::io::{BufReader, Read};
use std::str;
use std::time::Instant;

#[derive(Default)]
pub struct ReturnType {
    node_idxs: Vec<i64>,
    f2p_nbr_idxs: Vec<i64>,
    table_name_idxs: Vec<i64>,
    col_name_idxs: Vec<i64>,
    sem_types: Vec<i64>,
    number_values: Vec<f32>,
    text_values: Vec<i64>,
    datetime_values: Vec<f32>,
    masks: Vec<bool>,
    is_targets: Vec<bool>,
}

#[pyclass]
pub struct Dataset {
    mmap: Mmap,
    offsets: Vec<i64>,
    ctx_len: i64,
    mask_prob: f64,
    dropout_tok: f64,
    dropout_f2p: f64,
    dropout_p2f: f64,
}

fn pad4<T: Clone>(v: &[T]) -> Vec<i64>
where
    i64: From<T>,
{
    let mut out = vec![i64::MAX; 4];
    for i in 0..v.len() {
        out[i] = v[i].clone().into();
    }
    out
}

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value = "rel-f1")]
    dataset_name: String,
    #[arg(default_value = "2048")]
    ctx_len: i64,
    #[arg(default_value = "0.4")]
    mask_prob: f64,
}

pub fn main(cli: Cli) {
    let tic = Instant::now();
    let dataset = Dataset::new(&cli.dataset_name, cli.ctx_len, cli.mask_prob, 0.0, 0.0, 0.0);
    println!("Dataset loaded in {:?}", tic.elapsed());

    let mut sum = 0;
    let mut sum_sq = 0;
    let num_trials = 1000;
    let mut rng = rand::rng();
    for _i in 0..num_trials {
        let tic = Instant::now();
        let _item =
            dataset.get_item_bfs(Some(rng.random_range(0..dataset.offsets.len() as i64 - 1)));
        let elapsed = tic.elapsed().as_micros();
        sum += elapsed;
        sum_sq += elapsed * elapsed;
    }
    let mean = sum as f64 / num_trials as f64;
    let std = (sum_sq as f64 / num_trials as f64 - mean * mean).sqrt();
    println!("Mean: {} us,\tStd: {} us", mean, std);
}

#[pymethods]
impl Dataset {
    #[new]
    fn new(
        dataset_name: &str,
        ctx_len: i64,
        mask_prob: f64,
        dropout_tok: f64,
        dropout_f2p: f64,
        dropout_p2f: f64,
    ) -> Self {
        let pre_path = format!("{}/scratch/pre/{}", var("HOME").unwrap(), dataset_name);
        let nodes_path = format!("{}/nodes.rkyv", pre_path);
        let file = fs::File::open(&nodes_path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        // let mmap = unsafe { MmapOptions::new().populate().map(&file).unwrap() };
        let offsets_path = format!("{}/offsets.rkyv", pre_path);
        let file = fs::File::open(&offsets_path).unwrap();
        let mut bytes = Vec::new();
        BufReader::new(file).read_to_end(&mut bytes).unwrap();
        // TODO: don't deserialize?
        let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes).unwrap();
        let offsets = rkyv::deserialize::<Offsets, Error>(archived).unwrap();
        let offsets = offsets.offsets;
        // TODO
        Self {
            mmap,
            offsets,
            ctx_len,
            mask_prob,
            dropout_tok,
            dropout_f2p,
            dropout_p2f,
        }
    }

    fn get_item_py<'a>(
        &self,
        py: Python<'a>,
        seed_node_idx: Option<i64>,
    ) -> Result<Bound<'a, PyTuple>, PyErr> {
        let out = self.get_item(seed_node_idx);
        (
            ("node_idxs", PyArray1::from_vec(py, out.node_idxs)),
            ("f2p_nbr_idxs", PyArray1::from_vec(py, out.f2p_nbr_idxs)),
            (
                "table_name_values",
                PyArray1::from_vec(py, out.table_name_idxs),
            ),
            ("col_name_values", PyArray1::from_vec(py, out.col_name_idxs)),
            ("sem_types", PyArray1::from_vec(py, out.sem_types)),
            ("number_values", PyArray1::from_vec(py, out.number_values)),
            ("text_values", PyArray1::from_vec(py, out.text_values)),
            (
                "datetime_values",
                PyArray1::from_vec(py, out.datetime_values),
            ),
            ("masks", PyArray1::from_vec(py, out.masks)),
            ("is_targets", PyArray1::from_vec(py, out.is_targets)),
        )
            .into_pyobject(py)
    }
}

impl Dataset {
    // TODO: revert to non-offset implementation for speed
    fn get_node(&self, idx: i64) -> &ArchivedNode {
        let l = self.offsets[idx as usize] as usize;
        let r = self.offsets[(idx + 1) as usize] as usize;
        let bytes = &self.mmap[l..r];
        // TODO: access unchecked
        rkyv::access::<ArchivedNode, Error>(bytes).unwrap()
        // unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) }
    }

    // TODO: there is redundancy here with get_item_bfs
    // things like dropout don't work in one
    fn get_item(&self, seed_node_idx: Option<i64>) -> ReturnType {
        // let instant = Instant::now();
        let mut rng = rand::rng();

        let mut out = ReturnType::default();

        let mut visited = vec![false; self.offsets.len() - 1];

        let mut f2p_ftr = vec![];
        let seed_node = match seed_node_idx {
            Some(seed_node_idx) => {
                f2p_ftr.push((0, seed_node_idx));
                Some(self.get_node(seed_node_idx))
            }
            None => None,
        };
        let mut p2f_ftr = Vec::<Vec<_>>::new();

        loop {
            // dbg!(f2p_ftr.len());
            // dbg!(p2f_ftr.iter().map(|x| x.len()).collect::<Vec<_>>());

            // select node
            let (depth, node_idx) = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else {
                let mut depth_choices = Vec::new();
                for i in 0..p2f_ftr.len() {
                    if !p2f_ftr[i].is_empty() {
                        depth_choices.push(i);
                    }
                }
                if depth_choices.is_empty() {
                    // println!("Graph exhausted after {} cells", out.node_idxs.len());
                    // panic!();
                    (0, rng.random_range(0..self.offsets.len() - 1) as i64)
                } else {
                    // let depth = depth_choices[rng.random_range(0..depth_choices.len())];
                    let depth = depth_choices[0];
                    let r = rng.random_range(0..p2f_ftr[depth].len());
                    let l = p2f_ftr[depth].len();
                    let tmp = p2f_ftr[depth][r];
                    p2f_ftr[depth][r] = p2f_ftr[depth][l - 1];
                    p2f_ftr[depth][l - 1] = tmp;
                    let node_idx = p2f_ftr[depth].pop().unwrap();
                    (depth, node_idx)
                }
            };

            if visited[node_idx as usize] {
                continue;
            }
            visited[node_idx as usize] = true;

            let node = self.get_node(node_idx);

            for edge in node.f2p_edges.iter() {
                // dbg!(&edge);
                f2p_ftr.push((depth + 1, edge.node_idx.into()));
            }

            for edge in node.p2f_edges.iter() {
                match seed_node {
                    Some(seed_node) => {
                        if edge.timestamp.is_some()
                            && seed_node.timestamp.is_some()
                            && edge.timestamp > seed_node.timestamp
                        {
                            continue;
                        }
                        if edge.table_type != ArchivedTableType::Db
                            && edge.table_name_idx != seed_node.table_name_idx
                        {
                            continue;
                        }
                    }
                    None => {
                        if edge.table_type != ArchivedTableType::Db {
                            continue;
                        }
                    }
                }
                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                // dbg!(&edge);
                p2f_ftr[depth + 1].push(edge.node_idx.into());
            }

            let num_cells = node.col_name_idxs.len();
            for i in 0..num_cells {
                if rng.random_bool(self.dropout_tok) {
                    continue;
                }
                out.node_idxs.push(node.node_idx.into());
                out.f2p_nbr_idxs.extend(pad4(node.f2p_nbr_idxs.as_slice()));
                out.table_name_idxs.push(node.table_name_idx.into());
                out.col_name_idxs.push(node.col_name_idxs[i].into());
                let s = node.sem_types[i].clone() as i64;
                out.sem_types.push(s);
                out.number_values.push(node.number_values[i].into());
                out.text_values.push(node.text_values[i].into());
                out.datetime_values.push(node.datetime_values[i].into());
                let mask = if let Some(seed_node_idx) = seed_node_idx {
                    (s == SemType::Number as i64 && seed_node_idx == node.node_idx) ||
                        (rng.random_bool(self.mask_prob))

                } else {
                    rng.random_bool(self.mask_prob)
                };
                out.masks.push(mask);
                let is_target = if let Some(seed_node_idx) = seed_node_idx {
                    s == SemType::Number as i64 && seed_node_idx == node.node_idx
                } else {
                    false
                };
                out.is_targets.push(is_target);
                if out.node_idxs.len() >= self.ctx_len as usize {
                    break;
                }
            }
            if out.node_idxs.len() >= self.ctx_len as usize {
                break;
            }

        }

        // println!("Time taken in get_item: {:?}", instant.elapsed());

        out
    }

    fn get_item_bfs(&self, seed_node_idx: Option<i64>) -> ReturnType {
        // let instant = Instant::now();

        let mut out = ReturnType::default();

        //TODO: should make this a hashset for scalability
        let mut visited = vec![false; self.offsets.len() - 1];
        // let mut visited = HashSet::new();

        let mut f2p_ftr = vec![];
        let seed_node = match seed_node_idx {
            Some(seed_node_idx) => {
                f2p_ftr.push(seed_node_idx);
                Some(self.get_node(seed_node_idx))
            }
            None => None,
        };
        let mut p2f_ftr = VecDeque::new();

        let mut rng = rand::rng();

        loop {
            // select node
            let node_idx = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else if !p2f_ftr.is_empty() {
                p2f_ftr.pop_front().unwrap()
            } else {
                // println!("BFS: Graph exhausted after {} cells", out.node_idxs.len());
                rng.random_range(0..self.offsets.len() - 1) as i64
            };

            if visited[node_idx as usize] {
                continue;
            }
            visited[node_idx as usize] = true;
            // if visited.contains(&node_idx) {
            //     continue;
            // }
            // visited.insert(node_idx);

            let node = self.get_node(node_idx);

            for edge in node.f2p_edges.iter() {
                if rng.random_bool(self.dropout_f2p) {
                    continue;
                }
                f2p_ftr.push(edge.node_idx.into());
            }

            for edge in node.p2f_edges.iter() {
                match seed_node {
                    Some(seed_node) => {
                        if edge.timestamp.is_some()
                            && seed_node.timestamp.is_some()
                            && edge.timestamp > seed_node.timestamp
                        {
                            continue;
                        }
                        if edge.table_type != ArchivedTableType::Db
                            && edge.table_name_idx != seed_node.table_name_idx
                        {
                            continue;
                        }
                        if rng.random_bool(self.dropout_p2f) {
                            continue;
                        }
                    }
                    None => {
                        if edge.table_type != ArchivedTableType::Db {
                            continue;
                        }
                    }
                }
                p2f_ftr.push_back(edge.node_idx.into());
            }

            let num_cells = node.col_name_idxs.len();
            for i in 0..num_cells {
                if rng.random_bool(self.dropout_tok) {
                    continue;
                }
                out.node_idxs.push(node.node_idx.into());
                out.f2p_nbr_idxs.extend(pad4(node.f2p_nbr_idxs.as_slice()));
                out.table_name_idxs.push(node.table_name_idx.into());
                out.col_name_idxs.push(node.col_name_idxs[i].into());
                let s = node.sem_types[i].clone() as i64;
                out.sem_types.push(s);
                out.number_values.push(node.number_values[i].into());
                out.text_values.push(node.text_values[i].into());
                out.datetime_values.push(node.datetime_values[i].into());
                let mask = if let Some(seed_node_idx) = seed_node_idx {
                    (s == SemType::Number as i64 && seed_node_idx == node.node_idx) ||
                        (rng.random_bool(self.mask_prob))

                } else {
                    rng.random_bool(self.mask_prob)
                };
                out.masks.push(mask);
                let is_target = if let Some(seed_node_idx) = seed_node_idx {
                    s == SemType::Number as i64 && seed_node_idx == node.node_idx
                } else {
                    false
                };
                out.is_targets.push(is_target);
                if out.node_idxs.len() >= self.ctx_len as usize {
                    break;
                }
            }
            if out.node_idxs.len() >= self.ctx_len as usize {
                break;
            }
        }

        // println!("Time taken in get_item: {:?}", instant.elapsed());

        out
    }
}

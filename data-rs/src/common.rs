use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

#[derive(SerdeSerialize, SerdeDeserialize)]
pub struct TableInfo {
    pub node_idx_offset: i64,
    pub num_nodes: i64,
}

#[derive(Archive, Deserialize, Serialize, Hash, Eq, PartialEq, Default, Clone, Debug)]
#[rkyv(derive(PartialEq, Debug))]
pub enum TableType {
    #[default]
    Db,
    Train,
    Val,
    Test,
}

#[derive(Archive, Deserialize, Serialize, Default)]
#[rkyv(derive(Debug))]
pub struct Edge {
    pub node_idx: i64,
    pub table_name_idx: i64,
    pub table_type: TableType,
    pub timestamp: Option<i64>,
}

#[derive(Archive, Deserialize, Serialize, Default)]
#[rkyv(derive(Clone, Debug))]
pub enum SemType {
    #[default]
    Number,
    Text,
    DateTime,
}

#[derive(Archive, Deserialize, Serialize, Default)]
pub struct Node {
    pub node_idx: i64,
    pub f2p_nbr_idxs: Vec<i64>,
    pub f2p_edges: Vec<Edge>,
    pub p2f_edges: Vec<Edge>,
    pub timestamp: Option<i64>,
    pub table_name_idx: i64,
    pub col_name_idxs: Vec<i64>,
    pub sem_types: Vec<SemType>,
    pub number_values: Vec<f32>,
    pub text_values: Vec<i64>,
    pub datetime_values: Vec<f32>,
}

#[derive(Archive, Deserialize, Serialize, Default)]
pub struct Offsets {
    pub offsets: Vec<i64>,
}

"""Optimizer-level batches confirmed from the fixed-order training logs."""

FIXED_BATCHES = {
    "B0": ["mapwise_china_map6_0D_t1_src0525", "mapwise_usa_map_33_t31_src2021",
           "mapwise_china_map21_0C_t41_src0646", "mapwise_china_map7_8D_t42_src0321"],
    "B1": ["mapwise_usa_map_4042_t21_src2823", "mapwise_china_map17_4D_t31_src0416",
           "mapwise_india_map131_6D_t42_src1769", "mapwise_usa_map_7_t18_src2131"],
    "B2": ["mapwise_usa_map_1136_t39_src2496", "mapwise_usa_map_689_t18_src2965",
           "mapwise_india_map10_8C_t16_src1115", "mapwise_china_map19_4D_t16_src0784"],
    "B3": ["mapwise_usa_map_6616_t35_src2376", "mapwise_china_map17_2D_t3_src0807",
           "mapwise_india_map121_2C_t21_src1715", "mapwise_china_map2_3D_t31_src0267"],
    "B4": ["mapwise_india_map139_1C_t18_src1814", "mapwise_china_map19_0C_t5_src0825",
           "mapwise_india_map1_0C_t18_src1008", "mapwise_usa_map_15912_t3_src2925"],
}


def index_fixed20(dataset):
    ids = list(dataset["qa_id"])
    expected = [qid for batch in FIXED_BATCHES.values() for qid in batch]
    if len(ids) != 20 or len(set(ids)) != 20 or set(ids) != set(expected):
        raise ValueError("Dataset must contain exactly the 20 declared QA IDs, once each")
    return {qid: i for i, qid in enumerate(ids)}

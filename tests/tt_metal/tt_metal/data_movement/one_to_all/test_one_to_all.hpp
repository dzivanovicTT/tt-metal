namespace tt::tt_metal::unit_tests::dm::core_to_all {

void directed_ideal_test(
    tt::ARCH arch_,
    std::vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_case_id,
    bool is_multicast,
    bool is_linked,
    CoreCoord mst_core_coord,
    CoreCoord sub_start_core_coord,
    CoreCoord sub_grid_size);

}

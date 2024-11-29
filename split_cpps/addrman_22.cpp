std::vector<std::pair<AddrInfo, AddressPosition>> AddrManImpl::GetEntries_(bool from_tried) const
{
    AssertLockHeld(cs);

    const int bucket_count = from_tried ? ADDRMAN_TRIED_BUCKET_COUNT : ADDRMAN_NEW_BUCKET_COUNT;
    std::vector<std::pair<AddrInfo, AddressPosition>> infos;
    for (int bucket = 0; bucket < bucket_count; ++bucket) {
        for (int position = 0; position < ADDRMAN_BUCKET_SIZE; ++position) {
            nid_type id = GetEntry(from_tried, bucket, position);
            if (id >= 0) {
                AddrInfo info = mapInfo.at(id);
                AddressPosition location = AddressPosition(
                    from_tried,
                    /*multiplicity_in=*/from_tried ? 1 : info.nRefCount,
                    bucket,
                    position);
                infos.emplace_back(info, location);
            }
        }
    }

    return infos;
}
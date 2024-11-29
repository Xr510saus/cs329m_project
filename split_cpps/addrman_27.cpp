std::optional<AddressPosition> AddrManImpl::FindAddressEntry_(const CAddress& addr)
{
    AssertLockHeld(cs);

    AddrInfo* addr_info = Find(addr);

    if (!addr_info) return std::nullopt;

    if(addr_info->fInTried) {
        int bucket{addr_info->GetTriedBucket(nKey, m_netgroupman)};
        return AddressPosition(/*tried_in=*/true,
                               /*multiplicity_in=*/1,
                               /*bucket_in=*/bucket,
                               /*position_in=*/addr_info->GetBucketPosition(nKey, false, bucket));
    } else {
        int bucket{addr_info->GetNewBucket(nKey, m_netgroupman)};
        return AddressPosition(/*tried_in=*/false,
                               /*multiplicity_in=*/addr_info->nRefCount,
                               /*bucket_in=*/bucket,
                               /*position_in=*/addr_info->GetBucketPosition(nKey, true, bucket));
    }
}
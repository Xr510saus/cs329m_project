nid_type AddrManImpl::GetEntry(bool use_tried, size_t bucket, size_t position) const
{
    AssertLockHeld(cs);

    if (use_tried) {
        if (Assume(position < ADDRMAN_BUCKET_SIZE) && Assume(bucket < ADDRMAN_TRIED_BUCKET_COUNT)) {
            return vvTried[bucket][position];
        }
    } else {
        if (Assume(position < ADDRMAN_BUCKET_SIZE) && Assume(bucket < ADDRMAN_NEW_BUCKET_COUNT)) {
            return vvNew[bucket][position];
        }
    }

    return -1;
}
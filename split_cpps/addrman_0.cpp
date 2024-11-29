int AddrInfo::GetTriedBucket(const uint256& nKey, const NetGroupManager& netgroupman) const
{
    uint64_t hash1 = (HashWriter{} << nKey << GetKey()).GetCheapHash();
    uint64_t hash2 = (HashWriter{} << nKey << netgroupman.GetGroup(*this) << (hash1 % ADDRMAN_TRIED_BUCKETS_PER_GROUP)).GetCheapHash();
    return hash2 % ADDRMAN_TRIED_BUCKET_COUNT;
}
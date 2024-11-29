int AddrInfo::GetNewBucket(const uint256& nKey, const CNetAddr& src, const NetGroupManager& netgroupman) const
{
    std::vector<unsigned char> vchSourceGroupKey = netgroupman.GetGroup(src);
    uint64_t hash1 = (HashWriter{} << nKey << netgroupman.GetGroup(*this) << vchSourceGroupKey).GetCheapHash();
    uint64_t hash2 = (HashWriter{} << nKey << vchSourceGroupKey << (hash1 % ADDRMAN_NEW_BUCKETS_PER_SOURCE_GROUP)).GetCheapHash();
    return hash2 % ADDRMAN_NEW_BUCKET_COUNT;
}
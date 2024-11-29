int AddrInfo::GetBucketPosition(const uint256& nKey, bool fNew, int bucket) const
{
    uint64_t hash1 = (HashWriter{} << nKey << (fNew ? uint8_t{'N'} : uint8_t{'K'}) << bucket << GetKey()).GetCheapHash();
    return hash1 % ADDRMAN_BUCKET_SIZE;
}
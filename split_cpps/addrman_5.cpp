AddrManImpl::AddrManImpl(const NetGroupManager& netgroupman, bool deterministic, int32_t consistency_check_ratio)
    : insecure_rand{deterministic}
    , nKey{deterministic ? uint256{1} : insecure_rand.rand256()}
    , m_consistency_check_ratio{consistency_check_ratio}
    , m_netgroupman{netgroupman}
{
    for (auto& bucket : vvNew) {
        for (auto& entry : bucket) {
            entry = -1;
        }
    }
    for (auto& bucket : vvTried) {
        for (auto& entry : bucket) {
            entry = -1;
        }
    }
}
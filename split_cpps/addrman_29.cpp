void AddrManImpl::Check() const
{
    AssertLockHeld(cs);

    // Run consistency checks 1 in m_consistency_check_ratio times if enabled
    if (m_consistency_check_ratio == 0) return;
    if (insecure_rand.randrange(m_consistency_check_ratio) >= 1) return;

    const int err{CheckAddrman()};
    if (err) {
        LogPrintf("ADDRMAN CONSISTENCY CHECK FAILED!!! err=%i\n", err);
        assert(false);
    }
}
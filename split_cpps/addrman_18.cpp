void AddrManImpl::Attempt_(const CService& addr, bool fCountFailure, NodeSeconds time)
{
    AssertLockHeld(cs);

    AddrInfo* pinfo = Find(addr);

    // if not found, bail out
    if (!pinfo)
        return;

    AddrInfo& info = *pinfo;

    // update info
    info.m_last_try = time;
    if (fCountFailure && info.m_last_count_attempt < m_last_good) {
        info.m_last_count_attempt = time;
        info.nAttempts++;
    }
}
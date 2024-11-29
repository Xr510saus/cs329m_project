void AddrManImpl::Connected_(const CService& addr, NodeSeconds time)
{
    AssertLockHeld(cs);

    AddrInfo* pinfo = Find(addr);

    // if not found, bail out
    if (!pinfo)
        return;

    AddrInfo& info = *pinfo;

    // update info
    const auto update_interval{20min};
    if (time - info.nTime > update_interval) {
        info.nTime = time;
    }
}
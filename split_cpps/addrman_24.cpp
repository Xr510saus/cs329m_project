void AddrManImpl::SetServices_(const CService& addr, ServiceFlags nServices)
{
    AssertLockHeld(cs);

    AddrInfo* pinfo = Find(addr);

    // if not found, bail out
    if (!pinfo)
        return;

    AddrInfo& info = *pinfo;

    // update info
    info.nServices = nServices;
}
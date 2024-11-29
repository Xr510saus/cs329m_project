AddrInfo* AddrManImpl::Find(const CService& addr, nid_type* pnId)
{
    AssertLockHeld(cs);

    const auto it = mapAddr.find(addr);
    if (it == mapAddr.end())
        return nullptr;
    if (pnId)
        *pnId = (*it).second;
    const auto it2 = mapInfo.find((*it).second);
    if (it2 != mapInfo.end())
        return &(*it2).second;
    return nullptr;
}
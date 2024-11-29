AddrInfo* AddrManImpl::Create(const CAddress& addr, const CNetAddr& addrSource, nid_type* pnId)
{
    AssertLockHeld(cs);

    nid_type nId = nIdCount++;
    mapInfo[nId] = AddrInfo(addr, addrSource);
    mapAddr[addr] = nId;
    mapInfo[nId].nRandomPos = vRandom.size();
    vRandom.push_back(nId);
    nNew++;
    m_network_counts[addr.GetNetwork()].n_new++;
    if (pnId)
        *pnId = nId;
    return &mapInfo[nId];
}
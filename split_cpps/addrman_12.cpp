void AddrManImpl::Delete(nid_type nId)
{
    AssertLockHeld(cs);

    assert(mapInfo.count(nId) != 0);
    AddrInfo& info = mapInfo[nId];
    assert(!info.fInTried);
    assert(info.nRefCount == 0);

    SwapRandom(info.nRandomPos, vRandom.size() - 1);
    m_network_counts[info.GetNetwork()].n_new--;
    vRandom.pop_back();
    mapAddr.erase(info);
    mapInfo.erase(nId);
    nNew--;
}
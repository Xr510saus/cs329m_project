bool AddrManImpl::Add_(const std::vector<CAddress>& vAddr, const CNetAddr& source, std::chrono::seconds time_penalty)
{
    int added{0};
    for (std::vector<CAddress>::const_iterator it = vAddr.begin(); it != vAddr.end(); it++) {
        added += AddSingle(*it, source, time_penalty) ? 1 : 0;
    }
    if (added > 0) {
        LogDebug(BCLog::ADDRMAN, "Added %i addresses (of %i) from %s: %i tried, %i new\n", added, vAddr.size(), source.ToStringAddr(), nTried, nNew);
    }
    return added > 0;
}
bool AddrManImpl::Add(const std::vector<CAddress>& vAddr, const CNetAddr& source, std::chrono::seconds time_penalty)
{
    LOCK(cs);
    Check();
    auto ret = Add_(vAddr, source, time_penalty);
    Check();
    return ret;
}
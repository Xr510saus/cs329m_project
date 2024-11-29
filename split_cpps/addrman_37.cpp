std::pair<CAddress, NodeSeconds> AddrManImpl::Select(bool new_only, const std::unordered_set<Network>& networks) const
{
    LOCK(cs);
    Check();
    auto addrRet = Select_(new_only, networks);
    Check();
    return addrRet;
}
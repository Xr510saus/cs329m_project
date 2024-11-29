std::pair<CAddress, NodeSeconds> AddrMan::Select(bool new_only, const std::unordered_set<Network>& networks) const
{
    return m_impl->Select(new_only, networks);
}
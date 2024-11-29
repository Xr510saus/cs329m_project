bool AddrMan::Add(const std::vector<CAddress>& vAddr, const CNetAddr& source, std::chrono::seconds time_penalty)
{
    return m_impl->Add(vAddr, source, time_penalty);
}
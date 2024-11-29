void AddrMan::Connected(const CService& addr, NodeSeconds time)
{
    m_impl->Connected(addr, time);
}
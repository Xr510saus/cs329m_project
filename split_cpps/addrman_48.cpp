bool AddrMan::Good(const CService& addr, NodeSeconds time)
{
    return m_impl->Good(addr, time);
}
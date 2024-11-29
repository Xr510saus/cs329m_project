void AddrMan::Attempt(const CService& addr, bool fCountFailure, NodeSeconds time)
{
    m_impl->Attempt(addr, fCountFailure, time);
}
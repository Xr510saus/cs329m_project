void AddrManImpl::Attempt(const CService& addr, bool fCountFailure, NodeSeconds time)
{
    LOCK(cs);
    Check();
    Attempt_(addr, fCountFailure, time);
    Check();
}
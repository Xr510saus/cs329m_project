void AddrManImpl::Connected(const CService& addr, NodeSeconds time)
{
    LOCK(cs);
    Check();
    Connected_(addr, time);
    Check();
}
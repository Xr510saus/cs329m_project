void AddrManImpl::SetServices(const CService& addr, ServiceFlags nServices)
{
    LOCK(cs);
    Check();
    SetServices_(addr, nServices);
    Check();
}
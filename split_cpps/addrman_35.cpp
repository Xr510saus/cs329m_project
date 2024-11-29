void AddrManImpl::ResolveCollisions()
{
    LOCK(cs);
    Check();
    ResolveCollisions_();
    Check();
}
std::pair<CAddress, NodeSeconds> AddrManImpl::SelectTriedCollision()
{
    LOCK(cs);
    Check();
    auto ret = SelectTriedCollision_();
    Check();
    return ret;
}
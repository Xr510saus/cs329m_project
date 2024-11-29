bool AddrManImpl::Good(const CService& addr, NodeSeconds time)
{
    LOCK(cs);
    Check();
    auto ret = Good_(addr, /*test_before_evict=*/true, time);
    Check();
    return ret;
}
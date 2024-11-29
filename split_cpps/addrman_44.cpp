void AddrMan::Serialize(Stream& s_) const
{
    m_impl->Serialize<Stream>(s_);
}
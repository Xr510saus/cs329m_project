void ThrowCatch()
{
    try
    {
        throw 1;
    }
    catch (...)
    {
        int temporary = 0;
    }
}
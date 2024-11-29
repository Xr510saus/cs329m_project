CScriptID ToScriptID(const ScriptHash& script_hash)
{
    return CScriptID{uint160{script_hash}};
}
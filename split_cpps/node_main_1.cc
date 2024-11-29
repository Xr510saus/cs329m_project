bool UnsetHijackableEnvs(base::Environment* env) {
  bool has = false;
  for (const char* name : kHijackableEnvs) {
    if (env->HasVar(name)) {
      env->UnSetVar(name);
      has = true;
    }
  }
  return has;
}
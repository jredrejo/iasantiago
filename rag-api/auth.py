import time, json, httpx, jwt
from fastapi import HTTPException, Request
from .settings import GOOGLE_CLIENT_ID, GOOGLE_OIDC_ISSUER, ALLOWED_EMAIL_DOMAIN

JWKS_CACHE = {"keys": None, "exp": 0}


async def get_google_jwks():
    global JWKS_CACHE
    if JWKS_CACHE["keys"] and JWKS_CACHE["exp"] > time.time():
        return JWKS_CACHE["keys"]
    async with httpx.AsyncClient(timeout=10) as client:
        oidc_conf = (
            await client.get(f"{GOOGLE_OIDC_ISSUER}/.well-known/openid-configuration")
        ).json()
        jwks_uri = oidc_conf["jwks_uri"]
        keys = (await client.get(jwks_uri)).json()
        JWKS_CACHE = {"keys": keys, "exp": time.time() + 3600}
        return keys


async def verify_bearer(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer")
    token = auth.split(" ", 1)[1]
    keys = await get_google_jwks()
    header = jwt.get_unverified_header(token)
    key = next((k for k in keys["keys"] if k["kid"] == header.get("kid")), None)
    if not key:
        raise HTTPException(status_code=401, detail="bad kid")
    try:
        payload = jwt.decode(
            token,
            jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key)),
            algorithms=[header["alg"]],
            audience=GOOGLE_CLIENT_ID,
            issuer=GOOGLE_OIDC_ISSUER,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
    email = payload.get("email", "")
    if not email.endswith(ALLOWED_EMAIL_DOMAIN):
        raise HTTPException(status_code=403, detail="forbidden domain")
    return payload


# endpoint para nginx auth_request (devuelve 200 si OK)
async def auth_request(request: Request):
    await verify_bearer(request)
    return {"ok": True}

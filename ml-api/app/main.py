from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request


from settings import config
from databases.connect import Session
from api.r_v1 import router_v1


# ++++++++++++++++++++++++++++++++++++++++++++ DEFINE APP +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
app = FastAPI(title=config.PROJECT_NAME, openapi_url="/api/openapi.json", docs_url="/api/docs", redoc_url="/api/redoc")

# ++++++++++++++++++++++++++++++++++++++++++++ ROUTER CONFIG ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
app.include_router(router_v1, prefix="/api/v1")

# ++++++++++++++++++++++++++++++++++++++++++++ CORS MIDDLEWARE ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ++++++++++++++++++++++++++++++++++++++++++++++ DB CONFIG ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    request.state.db = Session()
    response = await call_next(request)
    request.state.db.close()
    return response
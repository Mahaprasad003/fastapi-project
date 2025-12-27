from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import routes_auth, routes_predict
from app.core.exceptions import register_exception_handlers
from app.middleware.logging_middleware import LoggingMiddleware

app = FastAPI(title="Car Price Prediction API")

app.add_middleware(LoggingMiddleware)

# linking  endpoints
app.include_router(routes_auth.router, tags=["auth"])
app.include_router(routes_predict.router, tags=["Prediction"])

# monitoring using promehteus
Instrumentator().instrument(app).expose(app)

# add exception handler
register_exception_handlers(app)

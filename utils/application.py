import json
from typing import Type

import tornado.auth
import tornado.escape
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from agent_builder.utils.constants import LIVENESS_SERVICE, READINESS_SERVICE, SERVER_PORT
from prometheus_client import generate_latest, REGISTRY
from tornado.ioloop import IOLoop

class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        super().set_header("Content-Type", "application/json; charset=UTF-8")

class PodAppMetricsHandler(BaseHandler):
    def get(self):
        data = ""
        if REGISTRY is not None:
            data = generate_latest(REGISTRY)
            data = data.decode("utf-8")
        self.write(data)
        self.finish()


class AsyncApplication(object):
    """
    application for prediction
    """
    # set both health status to be false by default
    is_liveness = False
    is_readiness = False

    class ReadinessCheckHandler(BaseHandler):
        def get(self):
            self.write(self._get_health())

        def post(self):
            self.write(self._get_health())

        @staticmethod
        def _get_health():
            # global is_readiness
            if AsyncApplication.is_readiness is True:
                return {"status": "Success"}
            else:
                raise tornado.web.HTTPError(
                    status_code=500, log_message="Server is not Ready"
                )

    class LivenessCheckHandler(BaseHandler):
        def get(self):
            self.write(self._get_health())

        def post(self):
            self.write(self._get_health())

        @staticmethod
        def _get_health():
            if AsyncApplication.is_liveness is True:
                return {"status": "Success"}
            else:
                raise tornado.web.HTTPError(
                    status_code=500, log_message="Server is not healthy"
                )

    handlers = list(
        [
            (r"/readinessCheck", ReadinessCheckHandler),
            (r"/livenessCheck", LivenessCheckHandler),
            (r"/pod-app-metrics", PodAppMetricsHandler)
        ]
    )
    settings = dict(
        xsrf_cookies=False,
        debug=True,
        autoescape=None,
    )

    def add_handler(self, pattern: str, handler: Type[tornado.web.RequestHandler]):
        self.handlers.append((pattern, handler))

    def update_settings(self, **kwargs):
        self.settings.update(kwargs)

    def app_instance(self) -> tornado.web.Application:
        app = tornado.web.Application(handlers=self.handlers, **self.settings)
        # set both health status to be true when server is started
        self.set_health_status(LIVENESS_SERVICE, True)
        self.set_health_status(READINESS_SERVICE, True)
        return app

    def start(self, server_port: int = SERVER_PORT):
        app = self.app_instance()
        app.listen(server_port)
        IOLoop.current().start()

    def set_health_status(self, service: str, status: bool):
        assert service == LIVENESS_SERVICE or service == READINESS_SERVICE
        if service == LIVENESS_SERVICE:
            AsyncApplication.is_liveness = status
        else:
            AsyncApplication.is_readiness = status


class CorruptRequestError(RuntimeError):

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        if args:
            self.log_message = args[0]
        else:
            self.log_message = "no log message"
        self._code = kwargs.get("code", 404)

    @property
    def code(self):
        return self._code

    def __str__(self):
        return self.log_message

PACKAGE=aidu-ai-director

WEB_DIST_DIR=\
	src/aidu/ai/director/web/dist


SMOKE_MODULES=\
	aidu.ai.director.directors.math_tutor_director

include ../aidu-dev-tools/python-package.mk

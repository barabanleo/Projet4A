# -*- coding: utf-8 -*-
#
# This file is part of SIDEPIPE.
#
# SIDEPIPE is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEPIPE is a python package for acoustic parametrization for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEPIPE is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# SIDEPIPE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEPIPE.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2016 Anthony Larcher
"""




def coroutine(func):
    """
    Decorator that allows to forget about the first call of a coroutine .next()
    method or .send(None)
    This call is done inside the decorator
    :param func: the coroutine to decorate
    """
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        next(cr)
        return cr
    return start

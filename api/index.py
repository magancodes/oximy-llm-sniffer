"""Vercel serverless entrypoint.

Vercel's @vercel/python runtime serves the module-level `app` (a WSGI callable).
We put the repo root on sys.path so `import app` resolves app.py at the project
root, then re-export its Flask instance. vercel.json rewrites every path to this
one function, so Flask itself does all the routing: the page, /api/*, /static/*.

There is no tshark or other system binary involved: features.py parses pcaps in
pure Python (scapy), which is what makes serverless deployment possible.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402  Flask WSGI application, served by Vercel

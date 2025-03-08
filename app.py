from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import subprocess  # To run the facial recognition script
from datetime import datetime
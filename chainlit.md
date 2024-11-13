# Chainlit application configuration
[app]
title = "Document QA System"
description = "Upload a PDF document and ask questions about its contents."

# Theme customization
[theme]
primary_color = "#007bff"  # Bootstrap blue

# Chat window customization
[chat_window]
default_expand_messages = false
collapsed_messages = true

# Code syntax highlighting
[code]
theme = "github-dark"

# File upload configuration
[upload]
max_files = 1
max_size_mb = 20
accepted_files = [".pdf"]

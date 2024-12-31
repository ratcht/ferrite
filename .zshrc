export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
export LIBRARY_PATH="/opt/homebrew/opt/openblas/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/opt/openblas/lib:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/openblas/lib:$DYLD_LIBRARY_PATH"

# The next line updates PATH for the Google Cloud SDK.
if [ -f '/Users/alialh/Documents/google-cloud-sdk/path.zsh.inc' ]; then . '/Users/alialh/Documents/google-cloud-sdk/path.zsh.inc'; fi

# The next line enables shell command completion for gcloud.
if [ -f '/Users/alialh/Documents/google-cloud-sdk/completion.zsh.inc' ]; then . '/Users/alialh/Documents/google-cloud-sdk/completion.zsh.inc'; fi

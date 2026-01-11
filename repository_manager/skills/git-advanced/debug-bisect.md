Workflow: Find bug
1. `git_action` "git bisect start <bad> <good>".
2. Test, then "git bisect good/bad".
3. End: "git bisect reset".

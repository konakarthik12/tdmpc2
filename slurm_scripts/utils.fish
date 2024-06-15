#!/nfshomes/kkona/local/bin/fish


function testing
    echo "testinga"
end
function verify_changes_pushed
    set origin_url (git config --get remote.origin.url)

        # Check if the origin URL is not empty
    if test -z "$origin_url"
        echo "No origin URL found. Make sure you are inside a Git repository with a remote origin set."
        return 1
    end

    # Check for uncommitted changes
    set uncommitted_changes (git status --porcelain)
    if test -n "$uncommitted_changes"
        echo "You have uncommitted changes. Please commit or stash them before checking the remote commit."
        return 1
    end

    # Check for unpushed commits
    set unpushed_commits (git log --branches --not --remotes)
    if test -n "$unpushed_commits"
        echo "You have unpushed commits. Please push them before checking the remote commit."
        return 1
    end

    # Get the latest commit hash
    set latest_commit_hash (git rev-parse HEAD)


    # Extract the repository owner and name from the origin URL
    if string match -q "*github.com*" -- $origin_url
        if string match -q "git@github.com:*" -- $origin_url
            set owner_repo (string match -r 'git@github.com:([^/]+)/([^/]+)(.git)?$' -- $origin_url)
        else if string match -q "https://github.com/*" -- $origin_url
            set owner_repo (string match -r 'https://github.com/([^/]+)/([^/]+)(.git)?$' -- $origin_url)
        end
        set owner_repo (string join "/" $owner_repo[2] $owner_repo[3])

        # Remove the .git part if it exists
        set owner_repo (string replace -r '\.git$' '' -- $owner_repo)

    else
        echo "The URL is not a valid GitHub URL: $origin_url"
        exit 1
    end

    set api_url "https://api.github.com/repos/$owner_repo/commits/$latest_commit_hash"

    # Verify the latest commit hash exists on GitHub
    curl -s -o /dev/null -w "%{http_code}" $api_url | read -l status_code

    if ! test "$status_code" = "200"
        echo "The latest commit hash $latest_commit_hash does not exist on GitHub."
        exit 1
    end

    echo $latest_commit_hash
end


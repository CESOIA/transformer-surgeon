# Contributing Guidelines

## Branch Strategy

### Branch Types Explained
- **`main`**: Integration branch for features (protected)
  - Where all team features come together
  - Must be stable enough for others to base their work on
  - Receives updates through Pull Requests only

- **`draft-{username}`**: Personal development branches
  - Your personal "sandbox" for experimentation
  - Can contain broken code, work-in-progress, or experimental features
  - Only you should push to your own draft branch
  - Examples: `draft-tizio`, `draft-caio`, `draft-sempronio`

- **`feature-{name}`**: Temporary feature branches for PRs
  - Created when you're ready to propose changes to the team
  - Should contain clean, focused changes
  - Deleted after the PR is merged
  - Examples: `feature-compression-algorithm`, `feature-bug-fix-memory-leak`

### Workflow

#### 1. **Personal Development Phase**
   ```bash
   # Switch to your personal workspace
   git checkout draft-{your-username}
   
   # Get any updates from the remote (in case you work from multiple computers)
   git pull origin draft-{your-username}
   
   # Work on your changes - experiment freely!
   # You can break things, try different approaches, etc.
   
   # Save your work with descriptive commit messages
   git add . && git commit -m "Your changes"
   
   # Push your work to remote for backup/sharing between your devices
   git push origin draft-{your-username}
   ```
   **Purpose**: This is your safe space to experiment without affecting others.

#### 2. **Ready to Share Phase**
   ```bash
   # Switch to the team integration branch
   git checkout main
   
   # Get the latest team changes to avoid conflicts
   git pull origin dmainp
   
   # Create a clean feature branch for your proposal
   # Use descriptive names like: feature-new-compression, feature-fix-memory-leak
   git checkout -b feature-descriptive-name
   
   # Bring your personal work into the feature branch
   git merge draft-{your-username}
   
   # Push the feature branch to propose it to the team
   git push origin feature-descriptive-name
   
   # Go to GitHub and create a Pull Request: feature-descriptive-name → main
   # Add description, screenshots, testing notes, etc.
   ```
   **Purpose**: Create a clean, reviewable proposal for your changes.

#### 3. **After PR Merge Phase**
   ```bash
   # Switch back to the integration branch
   git checkout main
   
   # Get the updated main branch (now includes your merged changes)
   git pull origin main
   
   # Clean up: delete the temporary feature branch locally
   git branch -d feature-descriptive-name
   
   # Clean up: delete the temporary feature branch on remote
   git push origin --delete feature-descriptive-name
   
   # Optional: Update your draft branch with the latest main
   git checkout draft-{your-username}
   git merge main  # Brings team changes into your personal branch
   ```
   **Purpose**: Clean up temporary branches and stay current with team changes.

### Rules & Best Practices

#### Core Rules
- **Only push to your own `draft-{username}` branch**
  - Prevents conflicts and accidental overwrites of others' work
  - Your draft branch is your personal workspace

- **Keep personal drafts up-to-date with `main`**
  - Regularly merge main into your draft: `git merge main`
  - Prevents large merge conflicts later
  - Ensures you're working with the latest team changes

- **Use descriptive commit messages**
  - Good: "Add compression for vision transformer layers"
  - Bad: "fix stuff" or "wip"
  - Helps team understand project evolution

#### Commit Message Guidelines
```
# Good examples:
git commit -m "Add LinearLRD compression to Qwen2VL attention layers"
git commit -m "Fix memory leak in compression scheme manager"
git commit -m "Update README with new installation instructions"

# Bad examples:
git commit -m "fix"
git commit -m "changes"
git commit -m "wip"
```

### Branch Permissions & Expectations

- **`draft-{username}`**: Personal workspace
  - Only you should push here
  - Can contain experimental/broken code
  - Use for trying new ideas, debugging, learning

- **`main`**: Team integration and release branch
  - Requires Pull Request with team review
  - Should remain stable enough for others to base work on
  - All team members contribute here through PRs

### Example Workflow Scenario

```bash
# Starting new work on compression improvements
git checkout draft-tizio
git pull origin draft-tizio

# Work for several days, making multiple commits
git add . && git commit -m "Experiment with different compression ratios"
git push origin draft-tizio

git add . && git commit -m "Add support for pruning in attention layers"
git push origin draft-tizio

git add . && git commit -m "Fix bug in dimension calculation"
git push origin draft-tizio

# Ready to share with team
git checkout main
git pull origin main
git checkout -b feature-improved-attention-compression
git merge draft-tizio
git push origin feature-improved-attention-compression

# Create PR on GitHub: feature-improved-attention-compression → main
# Team reviews, suggests changes, approves

# After merge, clean up
git checkout main
git pull origin main
git branch -d feature-improved-attention-compression
git push origin --delete feature-improved-attention-compression

# Update personal branch with team changes
git checkout draft-tizio
git merge main
```

### Why This Workflow?

1. **🛡️ Safety**: Personal drafts protect you from breaking others' work
2. **🤝 Collaboration**: PRs enable code review and knowledge sharing  
3. **🧹 Clean History**: Temporary branches keep the repository organized
4. **🔄 Flexibility**: Experiment freely while maintaining team coordination
5. **⚡ Quality**: Multiple checkpoints ensure stable shared code

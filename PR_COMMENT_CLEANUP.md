# ✅ PR Comment Cleanup Implementation

## 🎯 Requirement

**One benchmark comment per PR** - Delete previous comments before posting new ones.

## 🔧 Implementation

### New Workflow Step

Added before posting the new comment:

```yaml
- name: Delete previous benchmark comments
  if: github.event_name == 'pull_request'
  continue-on-error: true
  uses: actions/github-script@v7
  with:
    script: |
      # 1. List all comments on the PR
      const comments = await github.rest.issues.listComments(...)
      
      # 2. Find comments from github-actions[bot] with benchmark marker
      for (const comment of comments.data) {
        if (comment.user.login === 'github-actions[bot]' &&
            comment.body.includes('📊 Benchmark Results')) {
          
          # 3. Delete the old comment
          await github.rest.issues.deleteComment(...)
        }
      }
```

### How It Works

```
┌─────────────────────────────────────┐
│ PR Workflow Execution               │
├─────────────────────────────────────┤
│ 1. Run benchmarks                   │
│ 2. Generate reports                 │
│ 3. Upload artifacts                 │
│ 4. Delete old comments ← NEW!       │
│    ├─ Find all PR comments          │
│    ├─ Filter by bot + marker text   │
│    └─ Delete matches                │
│ 5. Post new comment                 │
│    └─ Only one exists now ✓         │
└─────────────────────────────────────┘
```

## 🔍 Detection Logic

**Identifies benchmark comments by:**
1. **User:** `github-actions[bot]`
2. **Content marker:** Contains `📊 Benchmark Results`

**Result:** Only comments from this workflow are deleted.

## ✅ Benefits

### Before
```
PR Comments:
├─ 📊 Benchmark Results (1st run)
├─ 📊 Benchmark Results (2nd run)
├─ 📊 Benchmark Results (3rd run)
├─ Other review comments
└─ 📊 Benchmark Results (4th run)
```
**Problem:** Cluttered PR with multiple outdated results

### After
```
PR Comments:
├─ Other review comments
└─ 📊 Benchmark Results (latest only) ✓
```
**Solution:** Clean PR with only latest benchmark

## 🛡️ Safety Features

### 1. Continue on Error
```yaml
continue-on-error: true
```
- If deletion fails, workflow continues
- New comment still gets posted
- Doesn't block the build

### 2. PR-Only Execution
```yaml
if: github.event_name == 'pull_request'
```
- Only runs on PRs
- Push to main doesn't trigger deletion

### 3. Specific Filtering
```javascript
comment.user.login === 'github-actions[bot]'
comment.body.includes('📊 Benchmark Results')
```
- Only deletes bot's own comments
- Only deletes benchmark comments
- Won't touch other comments

## 📋 Behavior Examples

### Scenario 1: First PR Comment
```
Action: No previous comments found
Result: Post new comment (total: 1 comment)
```

### Scenario 2: Updating Existing Comment
```
Action: Find 1 previous benchmark comment
Result: Delete old, post new (total: 1 comment)
```

### Scenario 3: Multiple Old Comments
```
Action: Find 3 previous benchmark comments
Result: Delete all 3, post new (total: 1 comment)
```

### Scenario 4: Mixed Comments
```
Before:
├─ Review comment from user
├─ 📊 Benchmark Results (old)
└─ Another review comment

Action: Delete only benchmark comment

After:
├─ Review comment from user
├─ Another review comment
└─ 📊 Benchmark Results (new)
```

## 🧪 Testing

### Manual Test
```bash
# 1. Push change to PR
# 2. Wait for workflow to complete
# 3. Check PR comments - should have only 1 benchmark comment
# 4. Push another change
# 5. Wait for workflow
# 6. Check PR comments - still only 1 benchmark comment (updated)
```

### What to Verify
- ✅ Only one benchmark comment exists
- ✅ It's the latest result
- ✅ Other comments are untouched
- ✅ Workflow doesn't fail if deletion fails

## 🔐 Permissions

Already configured in workflow:
```yaml
permissions:
  contents: read
  pull-requests: write  # Needed for delete & create
  issues: write        # Issues API = PR comments
```

## ⚠️ Known Limitations

### 1. Race Conditions
If two workflow runs happen simultaneously:
- Both might try to post comments
- Might briefly have 2 comments
- Next run will clean up

**Impact:** Minimal, very rare

### 2. Manual Comments
If someone manually posts a comment with "📊 Benchmark Results":
- It will be deleted (matches the filter)

**Solution:** Use unique marker or check comment author more strictly

**Impact:** Unlikely, easily avoidable

## 🚀 Result

**Clean PRs with single, up-to-date benchmark comment!**

```
✓ No clutter from multiple benchmark runs
✓ Always shows latest results
✓ Easy to find current performance status
✓ Professional PR presentation
```

---

**Status:** ✅ Implemented and ready to test
**Priority:** High - Improves PR experience
**Impact:** Better PR readability and professionalism

def searchInsert(nums,target) :
    index = -1
    if not nums:
        return -1
    if nums[0] >target:
        return 0
    if nums[-1]<target:
        return len(nums)
    if nums[-1]==target:
        return len(nums)-1
    for i in range(len(nums)-1):
        if nums[i]==target:
            return i
        if nums[i]<target and nums[i+1]>target:
            return i + 1


n = [1,3,5,6]
t = 2
print(searchInsert(n,t))
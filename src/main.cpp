// File Storage - disk-based B+ tree implementation.
//
// The problem asks for a persistent key-value store (index -> multiple
// values) implemented as a file-backed database, with a very tight memory
// budget (5-6 MiB). We therefore keep at most a handful of fixed-size pages
// (4 KiB each) in memory at any time and perform all real storage on disk
// using pread/pwrite, rebuilding a classic B+ tree whose composite key is
// (index, value).
//
// Since the same (index, value) pair is guaranteed never to be duplicated,
// the tree is effectively a sorted *set* of composite keys - there is no
// extra payload to store beyond the key itself.
//
// Deletions simply remove the key from its leaf without any underflow
// rebalancing: because the total number of entries ever stored is bounded
// (<=100000), skipping rebalancing does not hurt correctness or blow up the
// tree height, it only slightly under-utilizes some leaves.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <fcntl.h>

static const int PAGE_SIZE = 4096;
static const int IDX_LEN = 64;
static const int MAX_LEAF = 60;
static const int MAX_INTERNAL = 56;

#pragma pack(push, 1)
struct Key {
    char idx[IDX_LEN];
    int32_t value;
};

struct LeafNode {
    int32_t isLeaf;
    int32_t numKeys;
    int32_t next; // page id of next leaf in sorted order, -1 if none
    Key keys[MAX_LEAF];
};

struct InternalNode {
    int32_t isLeaf;
    int32_t numKeys;
    int32_t next; // unused
    Key keys[MAX_INTERNAL];
    int32_t children[MAX_INTERNAL + 1];
};

struct Header {
    int32_t rootPageId;
    int32_t pageCount;
};
#pragma pack(pop)

static_assert(sizeof(LeafNode) <= PAGE_SIZE, "LeafNode too big");
static_assert(sizeof(InternalNode) <= PAGE_SIZE, "InternalNode too big");

static int fd;
static Header hdr;

static inline int keycmp(const Key &a, const Key &b) {
    int c = memcmp(a.idx, b.idx, IDX_LEN);
    if (c) return c;
    return (a.value > b.value) - (a.value < b.value);
}

static inline void readPage(int32_t pid, void *buf) {
    pread(fd, buf, PAGE_SIZE, (off_t)pid * PAGE_SIZE);
}
static inline void writePage(int32_t pid, const void *buf) {
    pwrite(fd, buf, PAGE_SIZE, (off_t)pid * PAGE_SIZE);
}
static inline void readHeader() {
    pread(fd, &hdr, sizeof(Header), 0);
}
static inline void writeHeader() {
    pwrite(fd, &hdr, sizeof(Header), 0);
}

static inline int32_t allocPage() {
    int32_t pid = hdr.pageCount;
    hdr.pageCount++;
    writeHeader();
    return pid;
}

static void initNewDB() {
    hdr.pageCount = 2; // page 0 = header, page 1 = root leaf
    hdr.rootPageId = 1;
    writeHeader();

    char buf[PAGE_SIZE];
    memset(buf, 0, PAGE_SIZE);
    LeafNode *leaf = reinterpret_cast<LeafNode *>(buf);
    leaf->isLeaf = 1;
    leaf->numKeys = 0;
    leaf->next = -1;
    writePage(1, buf);
}

struct SplitInfo {
    bool split = false;
    Key promoted{};
    int32_t newPage = -1;
};

static SplitInfo insertRec(int32_t pid, const Key &key) {
    char buf[PAGE_SIZE];
    readPage(pid, buf);
    int32_t isLeaf = *reinterpret_cast<int32_t *>(buf);
    SplitInfo result;

    if (isLeaf) {
        LeafNode *node = reinterpret_cast<LeafNode *>(buf);
        int n = node->numKeys;
        int pos = 0;
        while (pos < n && keycmp(node->keys[pos], key) < 0) pos++;
        if (pos < n && keycmp(node->keys[pos], key) == 0) {
            // Duplicate (index, value) - per problem guarantee this never
            // happens in valid input, but handle defensively as a no-op.
            return result;
        }

        if (n < MAX_LEAF) {
            for (int i = n; i > pos; i--) node->keys[i] = node->keys[i - 1];
            node->keys[pos] = key;
            node->numKeys = n + 1;
            writePage(pid, buf);
            return result;
        }

        // Overflow: split.
        Key tmp[MAX_LEAF + 1];
        for (int i = 0; i < pos; i++) tmp[i] = node->keys[i];
        tmp[pos] = key;
        for (int i = pos; i < n; i++) tmp[i + 1] = node->keys[i];
        int total = n + 1;
        int leftCount = total / 2;
        int rightCount = total - leftCount;

        node->numKeys = leftCount;
        for (int i = 0; i < leftCount; i++) node->keys[i] = tmp[i];

        int32_t newPid = allocPage();
        char rbuf[PAGE_SIZE];
        memset(rbuf, 0, PAGE_SIZE);
        LeafNode *rnode = reinterpret_cast<LeafNode *>(rbuf);
        rnode->isLeaf = 1;
        rnode->numKeys = rightCount;
        rnode->next = node->next;
        for (int i = 0; i < rightCount; i++) rnode->keys[i] = tmp[leftCount + i];

        node->next = newPid;
        writePage(pid, buf);
        writePage(newPid, rbuf);

        result.split = true;
        result.promoted = rnode->keys[0];
        result.newPage = newPid;
        return result;
    } else {
        InternalNode *node = reinterpret_cast<InternalNode *>(buf);
        int n = node->numKeys;
        int pos = 0;
        while (pos < n && keycmp(key, node->keys[pos]) >= 0) pos++;

        SplitInfo childSplit = insertRec(node->children[pos], key);
        if (!childSplit.split) {
            return result;
        }

        if (n < MAX_INTERNAL) {
            for (int i = n; i > pos; i--) {
                node->keys[i] = node->keys[i - 1];
                node->children[i + 1] = node->children[i];
            }
            node->keys[pos] = childSplit.promoted;
            node->children[pos + 1] = childSplit.newPage;
            node->numKeys = n + 1;
            writePage(pid, buf);
            return result;
        }

        // Overflow: split internal node.
        Key tmpK[MAX_INTERNAL + 1];
        int32_t tmpC[MAX_INTERNAL + 2];
        for (int i = 0; i < pos; i++) tmpK[i] = node->keys[i];
        tmpK[pos] = childSplit.promoted;
        for (int i = pos; i < n; i++) tmpK[i + 1] = node->keys[i];
        for (int i = 0; i <= pos; i++) tmpC[i] = node->children[i];
        tmpC[pos + 1] = childSplit.newPage;
        for (int i = pos + 1; i <= n; i++) tmpC[i + 1] = node->children[i];

        int totalK = n + 1;
        int mid = totalK / 2;
        int leftKeyCount = mid;
        int rightKeyCount = totalK - mid - 1;

        node->numKeys = leftKeyCount;
        for (int i = 0; i < leftKeyCount; i++) node->keys[i] = tmpK[i];
        for (int i = 0; i <= leftKeyCount; i++) node->children[i] = tmpC[i];

        int32_t newPid = allocPage();
        char rbuf[PAGE_SIZE];
        memset(rbuf, 0, PAGE_SIZE);
        InternalNode *rnode = reinterpret_cast<InternalNode *>(rbuf);
        rnode->isLeaf = 0;
        rnode->numKeys = rightKeyCount;
        rnode->next = -1;
        for (int i = 0; i < rightKeyCount; i++) rnode->keys[i] = tmpK[mid + 1 + i];
        for (int i = 0; i <= rightKeyCount; i++) rnode->children[i] = tmpC[mid + 1 + i];

        writePage(pid, buf);
        writePage(newPid, rbuf);

        result.split = true;
        result.promoted = tmpK[mid];
        result.newPage = newPid;
        return result;
    }
}

static void insertKey(const Key &key) {
    SplitInfo r = insertRec(hdr.rootPageId, key);
    if (r.split) {
        int32_t newRootId = allocPage();
        char buf[PAGE_SIZE];
        memset(buf, 0, PAGE_SIZE);
        InternalNode *root = reinterpret_cast<InternalNode *>(buf);
        root->isLeaf = 0;
        root->numKeys = 1;
        root->next = -1;
        root->keys[0] = r.promoted;
        root->children[0] = hdr.rootPageId;
        root->children[1] = r.newPage;
        writePage(newRootId, buf);
        hdr.rootPageId = newRootId;
        writeHeader();
    }
}

static void deleteKey(const Key &key) {
    int32_t pid = hdr.rootPageId;
    char buf[PAGE_SIZE];
    for (;;) {
        readPage(pid, buf);
        int32_t isLeaf = *reinterpret_cast<int32_t *>(buf);
        if (isLeaf) {
            LeafNode *node = reinterpret_cast<LeafNode *>(buf);
            int n = node->numKeys;
            int pos = 0;
            while (pos < n && keycmp(node->keys[pos], key) < 0) pos++;
            if (pos < n && keycmp(node->keys[pos], key) == 0) {
                for (int i = pos; i < n - 1; i++) node->keys[i] = node->keys[i + 1];
                node->numKeys = n - 1;
                writePage(pid, buf);
            }
            return;
        } else {
            InternalNode *node = reinterpret_cast<InternalNode *>(buf);
            int n = node->numKeys;
            int pos = 0;
            while (pos < n && keycmp(key, node->keys[pos]) >= 0) pos++;
            pid = node->children[pos];
        }
    }
}

static void findIndex(const char *s, int len) {
    Key search;
    memset(search.idx, 0, IDX_LEN);
    memcpy(search.idx, s, len);
    search.value = -1;

    int32_t pid = hdr.rootPageId;
    char buf[PAGE_SIZE];
    for (;;) {
        readPage(pid, buf);
        int32_t isLeaf = *reinterpret_cast<int32_t *>(buf);
        if (isLeaf) break;
        InternalNode *node = reinterpret_cast<InternalNode *>(buf);
        int n = node->numKeys;
        int pos = 0;
        while (pos < n && keycmp(search, node->keys[pos]) >= 0) pos++;
        pid = node->children[pos];
    }

    LeafNode *node = reinterpret_cast<LeafNode *>(buf);
    int n = node->numKeys;
    int pos = 0;
    while (pos < n && keycmp(node->keys[pos], search) < 0) pos++;

    bool any = false;
    for (;;) {
        if (pos >= n) {
            int32_t nxt = node->next;
            if (nxt == -1) break;
            readPage(nxt, buf);
            node = reinterpret_cast<LeafNode *>(buf);
            n = node->numKeys;
            pos = 0;
            continue;
        }
        if (memcmp(node->keys[pos].idx, search.idx, IDX_LEN) != 0) break;
        if (any) putchar(' ');
        printf("%d", node->keys[pos].value);
        any = true;
        pos++;
    }
    if (!any) printf("null");
    putchar('\n');
}

int main() {
    static char outbuf[1 << 16];
    setvbuf(stdout, outbuf, _IOFBF, sizeof(outbuf));

    const char *dbfile = "db.dat";
    bool exists = (access(dbfile, F_OK) == 0);
    fd = open(dbfile, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    if (!exists) {
        initNewDB();
    } else {
        readHeader();
    }

    static char line[300];
    if (!fgets(line, sizeof(line), stdin)) {
        close(fd);
        return 0;
    }
    long n = atol(line);

    for (long i = 0; i < n; i++) {
        if (!fgets(line, sizeof(line), stdin)) break;
        char *tok = strtok(line, " \t\r\n");
        if (!tok) continue;

        if (tok[0] == 'i') { // insert
            char *idxs = strtok(nullptr, " \t\r\n");
            char *vals = strtok(nullptr, " \t\r\n");
            if (!idxs || !vals) continue;
            Key k;
            memset(k.idx, 0, IDX_LEN);
            size_t len = strlen(idxs);
            if (len > IDX_LEN) len = IDX_LEN;
            memcpy(k.idx, idxs, len);
            k.value = (int32_t)strtol(vals, nullptr, 10);
            insertKey(k);
        } else if (tok[0] == 'd') { // delete
            char *idxs = strtok(nullptr, " \t\r\n");
            char *vals = strtok(nullptr, " \t\r\n");
            if (!idxs || !vals) continue;
            Key k;
            memset(k.idx, 0, IDX_LEN);
            size_t len = strlen(idxs);
            if (len > IDX_LEN) len = IDX_LEN;
            memcpy(k.idx, idxs, len);
            k.value = (int32_t)strtol(vals, nullptr, 10);
            deleteKey(k);
        } else if (tok[0] == 'f') { // find
            char *idxs = strtok(nullptr, " \t\r\n");
            if (!idxs) continue;
            size_t len = strlen(idxs);
            if (len > IDX_LEN) len = IDX_LEN;
            findIndex(idxs, (int)len);
        }
    }

    fflush(stdout);
    close(fd);
    return 0;
}

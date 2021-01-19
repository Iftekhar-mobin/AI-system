import sys
import time
import re


def seq_driver(scores, seq_range, search, content):
    match = []
    best_idx = []
    flag = False

    for ids, score in scores[:seq_range]:
        collect = []
        [collect.append(item) for key, item in
         CaptureSequence().seq_main(search, content[ids]) if key == 0]
        if collect:
            match.append([ids, sorted(collect, key=len)[-1], score])

    new_scores = sorted(match, key=(lambda tup: len(tup[1])), reverse=True)
    if len(search) - len(new_scores[0][1]) < 3:
        flag = True
        [best_idx.append((ids, score)) for ids, matches, score in new_scores]
        return best_idx, flag
    else:
        return scores, flag


def seq_charsToLines(seqs, lineArray):

    for i in range(len(seqs)):
        text = []
        for char in seqs[i][1]:
            text.append(lineArray[ord(char)])
        seqs[i] = (seqs[i][0], "".join(text))


def seq_linesToChars(text1, text2):

    lineArray = []
    lineHash = {}
    lineArray.append("")

    def seq_linesToCharsMunge(text):
        chars = []
        lineStart = 0
        lineEnd = -1
        while lineEnd < len(text) - 1:
            lineEnd = text.find("\n", lineStart)
            if lineEnd == -1:
                lineEnd = len(text) - 1
            line = text[lineStart: lineEnd + 1]

            if line in lineHash:
                chars.append(chr(lineHash[line]))
            else:
                if len(lineArray) == maxLines:
                    line = text[lineStart:]
                    lineEnd = len(text)
                lineArray.append(line)
                lineHash[line] = len(lineArray) - 1
                chars.append(chr(len(lineArray) - 1))
            lineStart = lineEnd + 1
        return "".join(chars)
    maxLines = 666666
    chars1 = seq_linesToCharsMunge(text1)
    maxLines = 1114111
    chars2 = seq_linesToCharsMunge(text2)
    return chars1, chars2, lineArray


class CaptureSequence:
    _NEED = re.compile(r"\n\r?\n$")
    _START = re.compile(r"^\r?\n\r?\n")
    _DELETE = -1
    _INSERT = 1
    _EQUAL = 0

    def __init__(self):
        self._Timeout = 1.0
        self._EditCost = 4

    def seq_main(self, text1, text2, checklines=True, deadline=None):
        if deadline is None:
            if self._Timeout <= 0:
                deadline = sys.maxsize
            else:
                deadline = time.time() + self._Timeout

        if text1 is None or text2 is None:
            pass

        if text1 == text2:
            if text1:
                return [(self._EQUAL, text1)]
            return []

        commonlength = self.seq_commonPrefix(text1, text2)
        commonprefix = text1[:commonlength]
        text1 = text1[commonlength:]
        text2 = text2[commonlength:]

        commonlength = self.seq_commonSuffix(text1, text2)
        if commonlength == 0:
            commonsuffix = ""
        else:
            commonsuffix = text1[-commonlength:]
            text1 = text1[:-commonlength]
            text2 = text2[:-commonlength]

        seqs = self.seq_compute(text1, text2, checklines, deadline)

        if commonprefix:
            seqs[:0] = [(self._EQUAL, commonprefix)]
        if commonsuffix:
            seqs.append((self._EQUAL, commonsuffix))
        self.seq_cleanupMerge(seqs)
        return seqs

    def seq_compute(self, text1, text2, checklines, deadline):

        if not text1:
            return [(self._INSERT, text2)]

        if not text2:
            return [(self._DELETE, text1)]

        if len(text1) > len(text2):
            (longtext, shorttext) = (text1, text2)
        else:
            (shorttext, longtext) = (text1, text2)
        i = longtext.find(shorttext)
        if i != -1:
            seqs = [
                (self._INSERT, longtext[:i]),
                (self._EQUAL, shorttext),
                (self._INSERT, longtext[i + len(shorttext):]),
            ]
            if len(text1) > len(text2):
                seqs[0] = (self._DELETE, seqs[0][1])
                seqs[2] = (self._DELETE, seqs[2][1])
            return seqs

        if len(shorttext) == 1:
            return [(self._DELETE, text1), (self._INSERT, text2)]

        hm = self.seq_halfMatch(text1, text2)
        if hm:
            (text1_a, text1_b, text2_a, text2_b, mid_common) = hm
            seqs_a = self.seq_main(text1_a, text2_a, checklines, deadline)
            seqs_b = self.seq_main(text1_b, text2_b, checklines, deadline)

            return seqs_a + [(self._EQUAL, mid_common)] + seqs_b

        if checklines and len(text1) > 100 and len(text2) > 100:
            return self.seq_lineMode(text1, text2, deadline)

        return self.seq_bisect(text1, text2, deadline)

    def seq_lineMode(self, text1, text2, deadline):

        (text1, text2, linearray) = seq_linesToChars(text1, text2)
        seqs = self.seq_main(text1, text2, False, deadline)

        seq_charsToLines(seqs, linearray)
        self.seq_cleanupSemantic(seqs)

        seqs.append((self._EQUAL, ""))
        pointer = 0
        count_delete = 0
        count_insert = 0
        text_delete = ""
        text_insert = ""
        while pointer < len(seqs):
            if seqs[pointer][0] == self._INSERT:
                count_insert += 1
                text_insert += seqs[pointer][1]
            elif seqs[pointer][0] == self._DELETE:
                count_delete += 1
                text_delete += seqs[pointer][1]
            elif seqs[pointer][0] == self._EQUAL:
                if count_delete >= 1 and count_insert >= 1:
                    subDiff = self.seq_main(text_delete, text_insert, False, deadline)
                    seqs[pointer - count_delete - count_insert: pointer] = subDiff
                    pointer = pointer - count_delete - count_insert + len(subDiff)
                count_insert = 0
                count_delete = 0
                text_delete = ""
                text_insert = ""

            pointer += 1

        seqs.pop()

        return seqs

    def seq_bisect(self, text1, text2, deadline):

        text1_length = len(text1)
        text2_length = len(text2)
        max_d = (text1_length + text2_length + 1) // 2
        v_offset = max_d
        v_length = 2 * max_d
        v1 = [-1] * v_length
        v1[v_offset + 1] = 0
        v2 = v1[:]
        delta = text1_length - text2_length
        front = delta % 2 != 0
        k1start = 0
        k1end = 0
        k2start = 0
        k2end = 0
        for d in range(max_d):
            if time.time() > deadline:
                break

            for k1 in range(-d + k1start, d + 1 - k1end, 2):
                k1_offset = v_offset + k1
                if k1 == -d or (k1 != d and v1[k1_offset - 1] < v1[k1_offset + 1]):
                    x1 = v1[k1_offset + 1]
                else:
                    x1 = v1[k1_offset - 1] + 1
                y1 = x1 - k1
                while x1 < text1_length and y1 < text2_length and text1[x1] == text2[y1]:
                    x1 += 1
                    y1 += 1
                v1[k1_offset] = x1
                if x1 > text1_length:
                    k1end += 2
                elif y1 > text2_length:
                    k1start += 2
                elif front:
                    k2_offset = v_offset + delta - k1
                    if 0 <= k2_offset < v_length and v2[k2_offset] != -1:
                        x2 = text1_length - v2[k2_offset]
                        if x1 >= x2:
                            return self.seq_bisectSplit(text1, text2, x1, y1, deadline)

            for k2 in range(-d + k2start, d + 1 - k2end, 2):
                k2_offset = v_offset + k2
                if k2 == -d or (k2 != d and v2[k2_offset - 1] < v2[k2_offset + 1]):
                    x2 = v2[k2_offset + 1]
                else:
                    x2 = v2[k2_offset - 1] + 1
                y2 = x2 - k2
                while (
                        x2 < text1_length
                        and y2 < text2_length
                        and text1[-x2 - 1] == text2[-y2 - 1]
                ):
                    x2 += 1
                    y2 += 1
                v2[k2_offset] = x2
                if x2 > text1_length:
                    k2end += 2
                elif y2 > text2_length:
                    k2start += 2
                elif not front:
                    k1_offset = v_offset + delta - k2
                    if 0 <= k1_offset < v_length and v1[k1_offset] != -1:
                        x1 = v1[k1_offset]
                        y1 = v_offset + x1 - k1_offset
                        x2 = text1_length - x2
                        if x1 >= x2:

                            return self.seq_bisectSplit(text1, text2, x1, y1, deadline)

        return [(self._DELETE, text1), (self._INSERT, text2)]

    def seq_bisectSplit(self, text1, text2, x, y, deadline):
        text1a = text1[:x]
        text2a = text2[:y]
        text1b = text1[x:]
        text2b = text2[y:]

        seqs = self.seq_main(text1a, text2a, False, deadline)
        seqsb = self.seq_main(text1b, text2b, False, deadline)

        return seqs + seqsb

    def seq_commonPrefix(self, text1, text2):
        if not text1 or not text2 or text1[0] != text2[0]:
            return 0
        pointermin = 0
        pointermax = min(len(text1), len(text2))
        pointermid = pointermax
        pointerstart = 0
        while pointermin < pointermid:
            if text1[pointerstart:pointermid] == text2[pointerstart:pointermid]:
                pointermin = pointermid
                pointerstart = pointermin
            else:
                pointermax = pointermid
            pointermid = (pointermax - pointermin) // 2 + pointermin
        return pointermid

    def seq_commonSuffix(self, text1, text2):
        if not text1 or not text2 or text1[-1] != text2[-1]:
            return 0

        pointermin = 0
        pointermax = min(len(text1), len(text2))
        pointermid = pointermax
        pointerend = 0
        while pointermin < pointermid:
            if text1[-pointermid: len(text1) - pointerend] == text2[-pointermid: len(text2) - pointerend]:
                pointermin = pointermid
                pointerend = pointermin
            else:
                pointermax = pointermid
            pointermid = (pointermax - pointermin) // 2 + pointermin
        return pointermid

    def seq_halfMatch(self, text1, text2):
        if self._Timeout <= 0:
            return None
        if len(text1) > len(text2):
            (longtext, shorttext) = (text1, text2)
        else:
            (shorttext, longtext) = (text1, text2)
        if len(longtext) < 4 or len(shorttext) * 2 < len(longtext):
            return None

        def seq_halfMatchI(longtext, shorttext, i):
            best_longtext_a = ''
            best_longtext_b = ''
            best_shorttext_a = ''
            best_shorttext_b = ''

            seed = longtext[i: i + len(longtext) // 4]
            best_common = ""
            j = shorttext.find(seed)
            while j != -1:
                prefixLength = self.seq_commonPrefix(longtext[i:], shorttext[j:])
                suffixLength = self.seq_commonSuffix(longtext[:i], shorttext[:j])
                if len(best_common) < suffixLength + prefixLength:
                    best_common = (shorttext[j - suffixLength: j] + shorttext[j: j + prefixLength])
                    best_longtext_a = longtext[: i - suffixLength]
                    best_longtext_b = longtext[i + prefixLength:]
                    best_shorttext_a = shorttext[: j - suffixLength]
                    best_shorttext_b = shorttext[j + prefixLength:]
                j = shorttext.find(seed, j + 1)

            if len(best_common) * 2 >= len(longtext):
                return best_longtext_a, best_longtext_b, best_shorttext_a, best_shorttext_b, best_common
            else:
                return None

        hm1 = seq_halfMatchI(longtext, shorttext, (len(longtext) + 3) // 4)
        hm2 = seq_halfMatchI(longtext, shorttext, (len(longtext) + 1) // 2)
        if not hm1 and not hm2:
            return None
        elif not hm2:
            hm = hm1
        elif not hm1:
            hm = hm2
        else:
            if len(hm1[4]) > len(hm2[4]):
                hm = hm1
            else:
                hm = hm2

        if len(text1) > len(text2):
            (text1_a, text1_b, text2_a, text2_b, mid_common) = hm
        else:
            (text2_a, text2_b, text1_a, text1_b, mid_common) = hm
        return text1_a, text1_b, text2_a, text2_b, mid_common

    def seq_cleanupMerge(self, seqs):
        seqs.append((self._EQUAL, ""))
        pointer = 0
        count_delete = 0
        count_insert = 0
        text_delete = ""
        text_insert = ""
        while pointer < len(seqs):
            if seqs[pointer][0] == self._INSERT:
                count_insert += 1
                text_insert += seqs[pointer][1]
                pointer += 1
            elif seqs[pointer][0] == self._DELETE:
                count_delete += 1
                text_delete += seqs[pointer][1]
                pointer += 1
            elif seqs[pointer][0] == self._EQUAL:
                if count_delete + count_insert > 1:
                    if count_delete != 0 and count_insert != 0:
                        commonlength = self.seq_commonPrefix(text_insert, text_delete)
                        if commonlength != 0:
                            x = pointer - count_delete - count_insert - 1
                            if x >= 0 and seqs[x][0] == self._EQUAL:
                                seqs[x] = (seqs[x][0], seqs[x][1] + text_insert[:commonlength],)
                            else:
                                seqs.insert(0, (self._EQUAL, text_insert[:commonlength]))
                                pointer += 1
                            text_insert = text_insert[commonlength:]
                            text_delete = text_delete[commonlength:]

                        commonlength = self.seq_commonSuffix(text_insert, text_delete)
                        if commonlength != 0:
                            seqs[pointer] = (seqs[pointer][0], text_insert[-commonlength:] + seqs[pointer][1],)
                            text_insert = text_insert[:-commonlength]
                            text_delete = text_delete[:-commonlength]

                    new_ops = []
                    if len(text_delete) != 0:
                        new_ops.append((self._DELETE, text_delete))
                    if len(text_insert) != 0:
                        new_ops.append((self._INSERT, text_insert))
                    pointer -= count_delete + count_insert
                    seqs[pointer: pointer + count_delete + count_insert] = new_ops
                    pointer += len(new_ops) + 1
                elif pointer != 0 and seqs[pointer - 1][0] == self._EQUAL:
                    seqs[pointer - 1] = (
                        seqs[pointer - 1][0],
                        seqs[pointer - 1][1] + seqs[pointer][1],
                    )
                    del seqs[pointer]
                else:
                    pointer += 1

                count_insert = 0
                count_delete = 0
                text_delete = ""
                text_insert = ""

        if seqs[-1][1] == "":
            seqs.pop()

        changes = False
        pointer = 1

        while pointer < len(seqs) - 1:
            if seqs[pointer - 1][0] == self._EQUAL and seqs[pointer + 1][0] == self._EQUAL:
                if seqs[pointer][1].endswith(seqs[pointer - 1][1]):
                    if seqs[pointer - 1][1] != "":
                        seqs[pointer] = (seqs[pointer][0], seqs[pointer - 1][1] +
                                          seqs[pointer][1][: -len(seqs[pointer - 1][1])],)
                        seqs[pointer + 1] = (seqs[pointer + 1][0], seqs[pointer - 1][1] + seqs[pointer + 1][1],)
                    del seqs[pointer - 1]
                    changes = True
                elif seqs[pointer][1].startswith(seqs[pointer + 1][1]):

                    seqs[pointer - 1] = (seqs[pointer - 1][0], seqs[pointer - 1][1] + seqs[pointer + 1][1],)
                    seqs[pointer] = (seqs[pointer][0], seqs[pointer][1][len(seqs[pointer + 1][1]):] + seqs[pointer + 1][1],)
                    del seqs[pointer + 1]
                    changes = True
            pointer += 1

        if changes:
            self.seq_cleanupMerge(seqs)

    def seq_cleanupSemantic(self, seqs):

        changes = False
        equalities = []
        lastEquality = None
        pointer = 0
        length_insertions1, length_deletions1 = 0, 0
        length_insertions2, length_deletions2 = 0, 0
        while pointer < len(seqs):
            if seqs[pointer][0] == self._EQUAL:  # Equality found.
                equalities.append(pointer)
                length_insertions1, length_insertions2 = length_insertions2, 0
                length_deletions1, length_deletions2 = length_deletions2, 0
                lastEquality = seqs[pointer][1]
            else:  # An insertion or deletion.
                if seqs[pointer][0] == self._INSERT:
                    length_insertions2 += len(seqs[pointer][1])
                else:
                    length_deletions2 += len(seqs[pointer][1])

                if (lastEquality and (len(lastEquality) <= max(length_insertions1, length_deletions1))
                        and (len(lastEquality) <= max(length_insertions2, length_deletions2))):

                    seqs.insert(equalities[-1], (self._DELETE, lastEquality))
                    seqs[equalities[-1] + 1] = (self._INSERT, seqs[equalities[-1] + 1][1],)
                    equalities.pop()
                    if len(equalities):
                        equalities.pop()
                    if len(equalities):
                        pointer = equalities[-1]
                    else:
                        pointer = -1
                    # Reset the counters.
                    length_insertions1, length_deletions1 = 0, 0
                    length_insertions2, length_deletions2 = 0, 0
                    lastEquality = None
                    changes = True
            pointer += 1

        if changes:
            self.seq_cleanupMerge(seqs)
        self.seq_cleanupSemanticLossless(seqs)

        pointer = 1
        while pointer < len(seqs):
            if seqs[pointer - 1][0] == self._DELETE and seqs[pointer][0] == self._INSERT:
                deletion = seqs[pointer - 1][1]
                insertion = seqs[pointer][1]
                overlap_length1 = self.seq_commonOverlap(deletion, insertion)
                overlap_length2 = self.seq_commonOverlap(insertion, deletion)
                if overlap_length1 >= overlap_length2:
                    if overlap_length1 >= len(deletion) / 2.0 or overlap_length1 >= len(insertion) / 2.0:
                        seqs.insert(pointer, (self._EQUAL, insertion[:overlap_length1]))
                        seqs[pointer - 1] = (self._DELETE, deletion[: len(deletion) - overlap_length1],)
                        seqs[pointer + 1] = (self._INSERT, insertion[overlap_length1:],)
                        pointer += 1
                else:
                    if overlap_length2 >= len(deletion) / 2.0 or overlap_length2 >= len(insertion) / 2.0:
                        seqs.insert(pointer, (self._EQUAL, deletion[:overlap_length2]))
                        seqs[pointer - 1] = (self._INSERT, insertion[: len(insertion) - overlap_length2],)
                        seqs[pointer + 1] = (self._DELETE, deletion[overlap_length2:],)
                        pointer += 1
                pointer += 1
            pointer += 1

    def seq_cleanupSemanticLossless(self, seqs):
        def seq_cleanupSemanticScore(one, two):
            if not one or not two:
                return 6

            char1 = one[-1]
            char2 = two[0]
            nonAlphaNumeric1 = not char1.isalnum()
            nonAlphaNumeric2 = not char2.isalnum()
            whitespace1 = nonAlphaNumeric1 and char1.isspace()
            whitespace2 = nonAlphaNumeric2 and char2.isspace()
            lineBreak1 = whitespace1 and (char1 == "\r" or char1 == "\n")
            lineBreak2 = whitespace2 and (char2 == "\r" or char2 == "\n")
            blankLine1 = lineBreak1 and self._NEED.search(one)
            blankLine2 = lineBreak2 and self._START.match(two)

            if blankLine1 or blankLine2:
                return 5
            elif lineBreak1 or lineBreak2:
                return 4
            elif nonAlphaNumeric1 and not whitespace1 and whitespace2:
                return 3
            elif whitespace1 or whitespace2:
                return 2
            elif nonAlphaNumeric1 or nonAlphaNumeric2:
                return 1
            return 0

        pointer = 1
        while pointer < len(seqs) - 1:
            if seqs[pointer - 1][0] == self._EQUAL and seqs[pointer + 1][0] == self._EQUAL:

                equality1 = seqs[pointer - 1][1]
                edit = seqs[pointer][1]
                equality2 = seqs[pointer + 1][1]

                commonOffset = self.seq_commonSuffix(equality1, edit)
                if commonOffset:
                    commonString = edit[-commonOffset:]
                    equality1 = equality1[:-commonOffset]
                    edit = commonString + edit[:-commonOffset]
                    equality2 = commonString + equality2

                bestEquality1 = equality1
                bestEdit = edit
                bestEquality2 = equality2
                bestScore = seq_cleanupSemanticScore(equality1, edit) + seq_cleanupSemanticScore(edit, equality2)
                while edit and equality2 and edit[0] == equality2[0]:
                    equality1 += edit[0]
                    edit = edit[1:] + equality2[0]
                    equality2 = equality2[1:]
                    score = seq_cleanupSemanticScore(equality1, edit) + seq_cleanupSemanticScore(edit, equality2)

                    if score >= bestScore:
                        bestScore = score
                        bestEquality1 = equality1
                        bestEdit = edit
                        bestEquality2 = equality2

                if seqs[pointer - 1][1] != bestEquality1:

                    if bestEquality1:
                        seqs[pointer - 1] = (seqs[pointer - 1][0], bestEquality1)
                    else:
                        del seqs[pointer - 1]
                        pointer -= 1
                    seqs[pointer] = (seqs[pointer][0], bestEdit)
                    if bestEquality2:
                        seqs[pointer + 1] = (seqs[pointer + 1][0], bestEquality2)
                    else:
                        del seqs[pointer + 1]
                        pointer -= 1
            pointer += 1



    def seq_commonOverlap(self, text1, text2):
        text1_length = len(text1)
        text2_length = len(text2)

        if text1_length == 0 or text2_length == 0:
            return 0

        if text1_length > text2_length:
            text1 = text1[-text2_length:]
        elif text1_length < text2_length:
            text2 = text2[:text1_length]
        text_length = min(text1_length, text2_length)

        if text1 == text2:
            return text_length

        best = 0
        length = 1
        while True:
            pattern = text1[-length:]
            found = text2.find(pattern)
            if found == -1:
                return best
            length += found
            if found == 0 or text1[-length:] == text2[:length]:
                best = length
                length += 1

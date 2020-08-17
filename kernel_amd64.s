//+build !noasm,amd64

// func mul(x *[4]float32, y *[4]float32, result *[4]float32)
TEXT ·mul(SB),4,$0-24
	MOVQ x+0(FP), AX

	MOVQ y+8(FP), BX

	MOVQ result+16(FP), CX

	BYTE $0xC5; BYTE $0xF8; BYTE $0x10; BYTE $0x00 // VMOVUPS xmm0, [rax]

	BYTE $0xC5; BYTE $0xF8; BYTE $0x10; BYTE $0x0B // VMOVUPS xmm1, [rbx]

	BYTE $0xC5; BYTE $0xF8; BYTE $0x59; BYTE $0xD1 // VMULPS xmm2, xmm0, xmm1

	BYTE $0xC5; BYTE $0xF8; BYTE $0x11; BYTE $0x11 // VMOVUPS [rcx], xmm2

	BYTE $0xC5; BYTE $0xF8; BYTE $0x77 // VZEROUPPER

	RET

// func add(x *[4]float32, y *[4]float32, result *[4]float32)
TEXT ·add(SB),4,$0-24
	MOVQ x+0(FP), AX

	MOVQ y+8(FP), BX

	MOVQ result+16(FP), CX

	BYTE $0xC5; BYTE $0xF8; BYTE $0x10; BYTE $0x00 // VMOVUPS xmm0, [rax]

	BYTE $0xC5; BYTE $0xF8; BYTE $0x10; BYTE $0x0B // VMOVUPS xmm1, [rbx]

	BYTE $0xC5; BYTE $0xF8; BYTE $0x58; BYTE $0xD1 // VADDPS xmm2, xmm0, xmm1

	BYTE $0xC5; BYTE $0xF8; BYTE $0x11; BYTE $0x11 // VMOVUPS [rcx], xmm2

	BYTE $0xC5; BYTE $0xF8; BYTE $0x77 // VZEROUPPER

	RET

// func max(x *[4]float32, y *[4]float32, result *[4]float32)
TEXT ·max(SB),4,$0-24
	MOVQ x+0(FP), AX

	MOVQ y+8(FP), BX

	MOVQ result+16(FP), CX

	BYTE $0xC5; BYTE $0xF8; BYTE $0x10; BYTE $0x00 // VMOVUPS xmm0, [rax]

	BYTE $0xC5; BYTE $0xF8; BYTE $0x10; BYTE $0x0B // VMOVUPS xmm1, [rbx]

	BYTE $0xC5; BYTE $0xF8; BYTE $0x5F; BYTE $0xD1 // VMAXPS xmm2, xmm0, xmm1

	BYTE $0xC5; BYTE $0xF8; BYTE $0x11; BYTE $0x11 // VMOVUPS [rcx], xmm2

	BYTE $0xC5; BYTE $0xF8; BYTE $0x77 // VZEROUPPER

	RET

// func min(x *[4]float32, y *[4]float32, result *[4]float32)
TEXT ·min(SB),4,$0-24
	MOVQ x+0(FP), AX

	MOVQ y+8(FP), BX

	MOVQ result+16(FP), CX

	BYTE $0xC5; BYTE $0xF8; BYTE $0x10; BYTE $0x00 // VMOVUPS xmm0, [rax]

	BYTE $0xC5; BYTE $0xF8; BYTE $0x10; BYTE $0x0B // VMOVUPS xmm1, [rbx]

	BYTE $0xC5; BYTE $0xF8; BYTE $0x5D; BYTE $0xD1 // VMINPS xmm2, xmm0, xmm1

	BYTE $0xC5; BYTE $0xF8; BYTE $0x11; BYTE $0x11 // VMOVUPS [rcx], xmm2

	BYTE $0xC5; BYTE $0xF8; BYTE $0x77 // VZEROUPPER

	RET

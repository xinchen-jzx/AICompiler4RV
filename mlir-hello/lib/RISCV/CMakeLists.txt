add_mlir_dialect_library(MLIRRISCV
		RISCVDialect.cpp
		RISCVOps.cpp
		LowerToAffine.cpp
		LowerToLLVM.cpp
		#ShapeInferencePass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/RISCV

        DEPENDS
        MLIRRISCVOpsIncGen
		#ShapeInferenceInterfaceIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)